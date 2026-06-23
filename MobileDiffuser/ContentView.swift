// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI

/// Creative-studio shell: three sections (Create / Library / Settings) as a tab bar on iOS and a
/// sidebar split on macOS, all sharing one `AppModel`. Respects the in-app appearance override
/// (which defaults to following the system scheme). Model management lives in Settings + the Create
/// toolbar, not as a top-level section.
struct ContentView: View {
    @State private var model = AppModel()

    var body: some View {
        shell
            .tint(Theme.accent)
            .background(Theme.bg)
            .preferredColorScheme(model.appearance.colorScheme)
    }

    @ViewBuilder private var shell: some View {
        #if os(macOS)
        NavigationSplitView {
            List(Tab.allCases, selection: $model.tab) { tab in
                Label(tab.title, systemImage: tab.icon).tag(tab)
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 200)
            .listStyle(.sidebar)
        } detail: {
            screen(for: model.tab)
                .frame(minWidth: 560, minHeight: 480)
        }
        #else
        TabView(selection: $model.tab) {
            ForEach(Tab.allCases) { tab in
                NavigationStack { screen(for: tab).navigationTitle(tab.title) }
                    .tabItem { Label(tab.title, systemImage: tab.icon) }
                    .tag(tab)
            }
        }
        #endif
    }

    @ViewBuilder private func screen(for tab: Tab) -> some View {
        switch tab {
        case .create: CreateView(model: model)
        case .library: LibraryView(model: model)
        case .settings: SettingsView(model: model)
        }
    }
}

#Preview { ContentView() }
